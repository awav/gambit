open Hlo

let find_matches ({ matcher; scorer; apply }) (Root tree) =
  let rec walk acc node parent =
    let acc = if (matcher node) then (scorer node, node, parent, apply node) :: acc else acc in
    match node with
    | Node { op = Parameter } -> acc
    | Node { op = Dot { lhs; rhs } } ->
      let acc = walk acc lhs node in
      walk acc rhs node
  in walk [] tree (Root tree)

let rec string_of_matches m =
  match m with
  | [] -> ""
  | (score, node, parent, rewrite) :: t ->
    let match_str = Printf.sprintf
      "(\n  %d,\n  %s,\n  %s,\n  %s\n),"
      score
      (string_of_hlo node)
      (string_of_hlo parent)
      (string_of_hlo rewrite)
    in
    match_str ^ (string_of_matches t)
