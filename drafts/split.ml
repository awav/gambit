(* Redo the best split algorithm *)

let primes = [2; 3; 5; 7; 11; 13; 17; 19; 23; 29; 31; 37; 41; 43; 47; 53; 59; 61; 67; 71; 73; 79; 83; 89; 97; 101; 103; 107; 109; 113; 127; 131; 137; 139; 149; 151; 157; 163; 167; 173; 179; 181; 191; 193; 197; 199; 211; 223; 227; 229; 233; 239; 241; 251; 257; 263; 269; 271; 277; 281; 283; 293; 307; 311; 313; 317; 331; 337; 347; 349; 353; 359; 367; 373; 379; 383; 389; 397; 401; 409; 419; 421; 431; 433; 439; 443; 449; 457; 461; 463; 467; 479; 487; 491; 499; 503; 509; 521; 523; 541; 547; 557; 563; 569; 571; 577; 587; 593; 599; 601; 607; 613; 617; 619; 631; 641; 643; 647; 653; 659; 661; 673; 677; 683; 691; 701; 709; 719; 727; 733; 739; 743; 751; 757; 761; 769; 773; 787; 797; 809; 811; 821; 823; 827; 829; 839; 853; 857; 859; 863; 877; 881; 883; 887; 907; 911; 919; 929; 937; 941; 947; 953; 967; 971; 977; 983; 991; 997]

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

let print_factors lst = ()

let _print_factors lst =
  Printf.printf "[\n";
  List.fold_left (fun () (p, n) -> Printf.printf "  %d -> %d\n" p n) () lst;
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
  print_factors (factors size);
  print_list splits;
  List.fold_left (fun acc fact -> if size / fact <= max_size && acc > fact then fact else acc) size splits

let size = 2000
let full_size_bytes = 2000 * 1000 * (64 / 8)
let max_intermediate_bytes = 134217728

let max_size = max_intermediate_bytes * size / full_size_bytes

let split = best_split size max_size
let () = Printf.printf "best split for size = %d; max = %d -> %d (part size %d)\n" size max_size split (size / split)
