open Hlo

(** [dfs_match_tree scorers root] is a list of all matches with
    scores and applied re-writes *)
let rec dfs_match_tree (rewrites:rewrite list) (hlo:hlo) : (int * hlo) list =
  let do_rewrite { test; score; apply } =
    if test hlo then Some (score hlo, apply hlo) else None in
  let alternatives = List.filter_map do_rewrite rewrites in
  match hlo with
  | Root root -> List.map (fun (s, r) -> s, Root r) (dfs_match_tree rewrites root)
  | Node { op = Parameter _ } -> alternatives
  | Node { op = Dot dot; shape; pristine; } ->
    let alt_lhs = dfs_match_tree rewrites dot.lhs in
    let lhs_alternatives =
      List.map (fun (scr, lhs) -> (scr, Node { shape; pristine; op = Dot { dot with lhs } } )) alt_lhs in
    let alt_rhs = dfs_match_tree rewrites dot.rhs in
    let rhs_alternatives =
      List.map (fun (scr, rhs) -> (scr, Node { shape; pristine; op = Dot { dot with rhs } } )) alt_rhs in
    alternatives @ lhs_alternatives @ rhs_alternatives

(** The simplest exploration strategy possible: Depth first search
    of the rewrite tree. *)
let rec dfs_rewrite rewrites hlo =
  match dfs_match_tree rewrites hlo with
  | [] -> (0, hlo)
  | alternatives ->
    List.fold_left
      (fun (prev_score, prev_hlo) (single_score, rewrite) ->
        let below_score, full_rewrite = dfs_rewrite rewrites rewrite in
        let full_score = below_score + single_score in
        if full_score < prev_score then (full_score, full_rewrite) else (prev_score, prev_hlo)
      )
      (0, hlo)
      alternatives
