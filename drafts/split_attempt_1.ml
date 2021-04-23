open Hlo

(* arbitrary at this point :P *)
let should_split node =
  let shape = shape_of node in
  List.fold_left ( * ) 1 shape > 999 * 1000

(* find a node which contracts a big tensor with a dot *)
let rec find node =
  match node with
  | Root root -> find root
  | Node { op; shape } ->
    match op with
    | Dot { lhs; rhs } when should_split lhs || should_split rhs -> Some node
    | _ -> List.find_map find (inputs_of node)

type split = {
  dim: int; (* dim to split *)
  split: hlo; (* expression to split *)
  join: hlo; (* contraction partner / expr not to split *)
  cs: int; (* contraction index for split *)
  cj: int; (* contraction index for join *)
  split_is_lhs: bool;
  orig_shape: shape;
}

(* check if the split ends in a dot we can split up *)
let rec check_can_terminate node =
  match node with
  | Root _ -> false
  | Node node ->
    match node.op with
    | Dot _ -> true
    | PointFn item -> check_can_terminate item
    | _ -> false

(* decide how to best split the inputs of the dot *)
(* TODO: Be more general with what splits are permissible / what will be attempted to be split *)
let split_dim (node: hlo) : split option =
  let Node node = node in
  let Dot dot = node.op in
  let max_dim shape ex =
    let (d, _ ) = shape
      |> List.mapi (fun i d -> (i, d))
      |> List.fold_left (fun (j, b) (i, d) -> if b < d && i <> ex then (i, d) else (j, b)) (-1, 0)
    in d
  in
  if should_split dot.lhs && check_can_terminate dot.lhs then
    Some {
      dim = max_dim (shape_of dot.lhs) dot.lhs_c;
      split = dot.lhs;
      join = dot.rhs;
      cs = dot.lhs_c;
      cj = dot.rhs_c;
      split_is_lhs = true;
      orig_shape = node.shape;
    }
  else if should_split dot.rhs && check_can_terminate dot.rhs then
    Some {
      dim = max_dim (shape_of dot.rhs) dot.rhs_c;
      split = dot.rhs;
      join = dot.rhs;
      cs = dot.rhs_c;
      cj = dot.lhs_c;
      split_is_lhs = false;
      orig_shape = node.shape;
    }
  else
    None

(* build HLO that splits into n parts *)
let rewrite split n =
  let orig_size = List.nth split.orig_shape split.dim in
  let size = orig_size / n in
  let replace_size shape =
    let f i d = if i = split.dim then size else d in
    List.mapi f shape
  in
  let rec part (Node node) idx =
    let make_slice (Node node) dim =
      let start =
        let f i d = if i = dim then idx * size else 0 in
        node.shape |> List.mapi f
      in
      let limit =
        let f i d = if i = dim then size else d in
        node.shape |> List.mapi f
      in
      Node { op = Slice { item = Node node; start; limit }; shape = node.shape |> replace_size; pristine = false }
    in
    match node.op with
    | PointFn item ->
      let item_part = part item idx in
      Node { op = PointFn item_part; shape = item |> shape_of |> replace_size; pristine = false }
    | Dot dot ->
      if split.dim < (rank_of dot.lhs) - 1 then
        (* split is in LHS *)
        let dim = if split.dim >= dot.lhs_c then split.dim + 1 else split.dim in
        let lhs = make_slice dot.lhs dim in
        make_dot lhs dot.rhs dot.lhs_c dot.rhs_c
      else
        (* split is in RHS *)
        let dim = split.dim - rank_of dot.lhs + 1 in
        let dim = if dim >= dot.rhs_c then dim + 1 else dim in
        let rhs = make_slice dot.rhs dim in
        make_dot dot.lhs rhs dot.lhs_c dot.rhs_c
  in
  let rec make_parts acc idx =
    if idx >= 0 then
      let p = part split.split idx in
      make_parts (p :: acc) (idx - 1)
    else acc in
  let parts = make_parts [] (n - 1) in
  (* make the top level dots for the parts *)
  let f part =
    if split.split_is_lhs then
      make_dot part split.join split.cs split.cj
    else
      make_dot split.join part split.cj split.cs
  in
  let dots = List.map f parts in
  Node { op = Concat { items = dots; dim = split.dim }; shape = split.orig_shape; pristine = true }


let rec pass node =
  match node with
  | Root root -> Root (pass root)
  | Node nd ->
    match nd.op with
    | Dot dot when should_split dot.lhs || should_split dot.rhs ->
      match split_dim node with
      | Some split -> rewrite split 2
      | None -> Node { nd with op = Dot { dot with lhs = pass dot.lhs; rhs = pass dot.rhs; }; }
    (* lazy for now ... *)


(* test case !!! *)

let a = Node { op = Parameter "a"; shape = [1000; 2]; pristine = true }
let a = Node { op = Concat { items = [a; a]; dim = 0 }; shape = [2000; 2]; pristine = true }

let b = Node { op = Parameter "b"; shape = [1000; 2]; pristine = true }
let c = Node { op = Parameter "c"; shape = [1000; 2]; pristine = true }

let ab = make_dot ~pristine:true a b 1 1
let p = Node { op = PointFn ab; shape = shape_of ab; pristine = true }

(* both of those should work ! *)
let pc = make_dot ~pristine:true p c 1 0
(* let pc = make_dot ~pristine:true c p 0 1 *)

let root = Root pc

let () = root |> string_of_hlo |> print_endline

(* Find an opportunity for optimization *)
let () = root |> pass |> string_of_hlo |> print_endline
