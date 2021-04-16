open Hlo
open Visitor

let elements_in shape =
  List.fold_left (fun acc d -> acc * d) 1 shape

let lhs_to_rhs node =
  match node with
  | Node {
      op = Dot {
        lhs = Node {
          op = Dot {
            lhs = Node a;
            rhs = Node b;
            lhs_c = contr_idx_a_with_b;
            rhs_c = contr_idx_b_with_a;
          };
          shape = inner_shape;
        };
        rhs = Node c;
        lhs_c = contr_idx_ab_with_c;
        rhs_c = contr_idx_c_with_ab;
      }
    } ->
      (*
      We want to rewrite (AB)C -> A(BC) or (AB)C -> B(AC)
      ===================================================
      Consider possible indexes involved:

         A         | B         | C
         ----------+-----------+-------
      1) 0 ab    n | 0 ba bc m | 0 cb l => ab; ba; abc = bc + rank(a) - 2; cab = cb;
      2) 0 ab    n | 0 bc ba m | 0 cb l => ab; ba; abc = bc + rank(a) - 1; cab = cb;
      3) 0 ab ac n | 0 ba    m | 0 ca l => ab; ba; abc = ac - 1; cab = ca;
      4) 0 ac ab n | 0 ba    m | 0 ca l => ab; ba; abc = ac; cab = ca;

      1) bc > 0 -> abc >= rank(a) - 1
      2) abc >= rank(a) - 1
      3) abc < ac && ac < rank(a) -> abc < rank(a) - 1
      4) ac < ab && abc = ac && ab < rank(a) -> abc < rank(a) - 1

      ==> can distinquish case as abc < rank(a) - 1 VS >= rank(a)-1
      *)
      let rank_a = List.length a.shape in
      let contr_idx_c_with_inner = contr_idx_c_with_ab in
      let (
        inner, (* either A or B, depending on which one goes with C *)
        outer, (* either B or A *)
        contr_idx_inner_with_outer,
        contr_idx_outer_with_inner,
        contr_idx_inner_with_c_unadjusted
      ) = if contr_idx_ab_with_c < rank_a - 1 then
        (* it's A with C *)
        (Node a, Node b, contr_idx_a_with_b, contr_idx_b_with_a, contr_idx_ab_with_c)
      else
        (* it's B with C *)
        (Node b, Node a, contr_idx_b_with_a, contr_idx_a_with_b, contr_idx_ab_with_c - (rank_a - 1))
      in
      (* this fixes cases when the previously inner index was before this one *)
      let contr_idx_inner_with_c = if contr_idx_inner_with_outer <= contr_idx_inner_with_c_unadjusted then
        contr_idx_inner_with_c_unadjusted + 1
      else
        contr_idx_inner_with_c_unadjusted
      in
      (*
      We now have
      outer  | inner     | c
      0 oi n | 0 io ic m | 0 ci l => ico = io     -- if inner_with_outer < inner_with_c
      0 oi n | 0 ic io m | 0 ci l => ico = io - 1 -- if inner_with_outer > inner_with_c
      *)
      let contr_idx_inner_c_with_outer = if contr_idx_inner_with_outer < contr_idx_inner_with_c then
        contr_idx_inner_with_outer
      else
        contr_idx_inner_with_outer - 1
      in
      let new_inner_dot = make_dot inner (Node c) contr_idx_inner_with_c contr_idx_c_with_inner in
      let new_outer_dot = make_dot outer new_inner_dot contr_idx_outer_with_inner contr_idx_inner_c_with_outer in
      (* Compute relative cost *)
      let current_size = elements_in inner_shape in
      let proposed_size = elements_in (match new_outer_dot with Node { shape } -> shape) in
      Some (proposed_size - current_size, new_outer_dot)
  | _ -> None

let lhs_to_rhs_matcher node =
  match node with
  | Node {
      op = Dot { lhs = Node { op = Dot _; pristine = inner_pristine } };
      pristine = outer_pristine
    } -> inner_pristine || outer_pristine
  | _ -> false

let lhs_to_rhs_rewrite = {
  matcher = lhs_to_rhs_matcher;
  scorer = (fun n -> match lhs_to_rhs n with Some (s, _) -> s | None -> 0);
  apply = (fun n -> match lhs_to_rhs n with Some (_, r) -> r | None -> n);
}


(* Test case! *)

let a = Node { op = Parameter "a"; shape = [1000; 2]; pristine = true }
let b = Node { op = Parameter "b"; shape = [1000; 2]; pristine = true }
let c = Node { op = Parameter "c"; shape = [1000; 2]; pristine = true }
let d = Node { op = Parameter "d"; shape = [1000; 2]; pristine = true }
let e = Node { op = Parameter "e"; shape = [1000; 2]; pristine = true }

let ab = make_dot ~pristine:true a b 1 1
let abc = make_dot ~pristine:true ab c 1 0
let abcd = make_dot ~pristine:true abc d 1 1
let abcde = make_dot ~pristine:true abcd e 1 0

let root = Root abcde

let () = print_endline "The full expression"
let () = root |> string_of_hlo |> print_endline

let () = print_endline "\nDepth first search with recursive replace at all"
let () = root
  |> (rewrite_dfs [lhs_to_rhs_rewrite])
  |> fun (score, tree) -> Printf.printf "Score: %d\n" score; tree
  |> string_of_hlo
  |> print_endline
