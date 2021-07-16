(* Redo the best split algorithm *)

let primes = [2; 3; 5; 7; 11; 13; 17]

let factors num =
  let factor prime =
    let rec loop n acc =
      if n mod prime = 0 then loop (n / prime) (acc + 1) else acc
    in prime, loop num 0
  in
  List.map factor primes

let print_list lst =
  Printf.printf "[";
  List.fold_left (fun () n -> Printf.printf "%d," n) () lst;
  Printf.printf "]\n"

let best_split size max_size =
  let rec comb curr acc = function
    | (prime, count) :: tail when count > 0 ->
      let curr' = prime * curr in
      let acc' = comb curr' (curr' :: acc) ((prime, count - 1) :: tail) in
      comb curr acc' tail
    | _ :: tail -> comb curr acc tail
    | [] -> acc
  in
  let splits = size |> factors |> comb 1 [] in
  print_list splits;
  List.fold_left (fun acc fact -> if size / fact <= max_size && acc > fact then fact else acc) size splits

let size = 100
let max_size = 30
let split = best_split size max_size
let () = Printf.printf "best split for size = %d; max = %d -> %d (part size %d)\n" size max_size split (size / split)
