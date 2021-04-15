open Hlo
open Visitor

let elements_in shape =
  List.fold_left (fun acc d -> acc * d) 1 shape

let lhs_to_rhs node =
  match node with
  | Node {
      op = Dot {
        lhs = Node {
          op = Dot { lhs = Node a; rhs = Node b; lhs_c = inner_lc; rhs_c = inner_rc };
          shape = inner_shape;
        };
        rhs = Node c;
        lhs_c = outer_lc;
        rhs_c = outer_rc;
      }
    } ->
      let current_size = elements_in inner_shape in
      let rank_a = List.length a.shape in
      (* TODO: Handle case of (AC)B (i.e. when the index inner is on A not B) *)
      (* Also, this is a mess, maybe I should draw some pictures or smth... *)
      let contr_idx_b_with_c = outer_lc - (rank_a - 1) in
      let contr_idx_a_with_bc = inner_lc in
      let contr_idx_bc_with_a = if contr_idx_b_with_c < inner_rc then inner_rc - 1 else inner_rc in
      let contr_idx_c_with_b = outer_rc in
      let proposed_size =
        (elements_in b.shape) / (List.nth b.shape contr_idx_b_with_c) *
        (elements_in c.shape) / (List.nth c.shape contr_idx_c_with_b) in
      let score = proposed_size - current_size in
      let new_inner_dot = make_dot (Node b) (Node c) contr_idx_b_with_c contr_idx_c_with_b in
      let new_node = make_dot (Node a) new_inner_dot contr_idx_a_with_bc contr_idx_bc_with_a in
      Some (score, new_node)
  | _ -> None

let lhs_to_rhs_rewrite = {
  matcher = (fun n -> match lhs_to_rhs n with Some _ -> true | None -> false);
  scorer = (fun n -> match lhs_to_rhs n with Some (s, _) -> s | None -> 0);
  apply = (fun n -> match lhs_to_rhs n with Some (_, r) -> r | None -> n);
}


(* Test case! *)

let a = Node { op = Parameter; shape = [1000; 2]; prestine = true }
let b = Node { op = Parameter; shape = [1000; 2]; prestine = true }
let c = Node { op = Parameter; shape = [1000; 2]; prestine = true }
let d = Node { op = Parameter; shape = [1000; 2]; prestine = true }
let e = Node { op = Parameter; shape = [1000; 2]; prestine = true }

let ab = make_dot a b 1 1
let abc = make_dot ab c 1 0
let abcd = make_dot abc d 1 1
let abcde = make_dot abcd e 1 0

let root = Root abc

let () = print_endline "The full expression"
let () = root |> string_of_hlo |> print_endline

let () = print_endline "\nAll rewrite matches and their scores!"
let () = root |> (find_matches lhs_to_rhs_rewrite) |> string_of_matches |> print_endline
