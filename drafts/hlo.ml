type shape = int list

and op =
  | Parameter of string
  | Dot of { lhs: hlo; rhs: hlo; lhs_c: int; rhs_c: int }
  | Concat of { items: hlo list; dim: int } (* HLOInstruction::CreateConcatenate *)
  | Slice of { item: hlo; start: shape; limit: shape } (* HLOInstruction::CreateSlice; stride is omitted *)
  | PointFn of hlo (* stand-in for any point-wise applied function like tf.math.sin *)

and hlo = Root of hlo | Node of { op: op; shape: shape; pristine: bool }

type rewrite = { test: hlo -> bool; score: hlo -> int; apply: hlo -> hlo }

let rec string_of_shape s =
  match s with
  | [] -> ""
  | d :: s -> Printf.sprintf "%d," d ^ string_of_shape s

let rec string_of_hlo ?ws:(ws="") node =
  match node with
  | Root root -> Printf.sprintf "Root {\n%s\n}" (string_of_hlo ~ws:"  " root)
  | Node node ->
    ws ^ (if node.pristine then "" else "*") ^ match node.op with
    | Parameter name -> Printf.sprintf "Parameter { '%s' shape=(%s) }" name (string_of_shape node.shape)
    | Dot { lhs; rhs; lhs_c; rhs_c } ->
      Printf.sprintf
        "Dot {\n  %slhs=\n%s;\n  %srhs=\n%s;\n  %s%d %d; shape=(%s)\n%s}"
        ws
        (string_of_hlo ~ws:("  "^ws) lhs)
        ws
        (string_of_hlo ~ws:("  "^ws) rhs)
        ws
        lhs_c rhs_c
        (string_of_shape node.shape)
        ws
    | Concat { items; dim } ->
      Printf.sprintf
        "Concat {\n%s  items=[%s\n%s  ]\n%s  dim=%d; shape=(%s)\n%s}"
        ws
        (
          items
          |> List.map (string_of_hlo ~ws:("    "^ws))
          |> List.map ( ( ^ ) "\n" )
          |> List.fold_left ( ^ ) ""
        )
        ws
        ws
        dim
        (string_of_shape node.shape)
        ws
    | Slice { item; start; limit } ->
      Printf.sprintf
        "Slice {\n%s  item=\n%s;\n%s  start=(%s); limit=(%s)\n%s}"
        ws
        (string_of_hlo ~ws:("  "^ws) item)
        ws
        (string_of_shape start)
        (string_of_shape limit)
        ws
    | PointFn item ->
      Printf.sprintf
        "PointFn {\n%s  item=\n%s;\n%s  shape=(%s)\n%s}"
        ws
        (string_of_hlo ~ws:("  "^ws) item)
        ws
        (string_of_shape node.shape)
        ws

let make_dot ?pristine:(pristine=false) lhs rhs i j =
  let rec out_shape sl sr i j so =
    match sl, sr, i, j with
    | _ :: sl, sr, 0, j -> out_shape sl sr (i - 1) j so
    | sl, _ :: sr, i, 0 -> out_shape sl sr i (j - 1) so
    | d :: sl, sr, i, j -> out_shape sl sr (i - 1) j (d :: so)
    | [], d :: sr, i, j -> out_shape sl sr i (j - 1) (d :: so)
    | [], [], _, _ -> List.rev so
  in match lhs, rhs with
  | Node lhs, Node rhs ->
    if List.nth lhs.shape i <> List.nth rhs.shape j then
      let () = Printf.printf
        "make_dot failed:\nlhs = %s\nrhs = %s\ni, j = %d, %d\n"
        (string_of_hlo (Node lhs))
        (string_of_hlo (Node rhs))
        i j in
      exit 1 else ();
    Node {
      op = Dot { lhs = Node lhs; rhs = Node rhs; lhs_c = i; rhs_c = j };
      shape = out_shape lhs.shape rhs.shape i j [];
      pristine;
    }
  | _ -> raise (Invalid_argument "need two nodes")

let rec shape_of node =
  match node with
  | Root root -> shape_of root
  | Node { shape } -> shape

let inputs_of node =
  match node with
  | Root root -> [root]
  | Node node ->
    match node.op with
    | Parameter _ -> []
    | Dot { lhs; rhs } -> [lhs; rhs]
    | Concat { items } -> items
    | Slice { item } -> [item]
    | PointFn item -> [item]

let rec rank_of = function
  | Root root -> rank_of root
  | Node { shape } -> List.length shape
