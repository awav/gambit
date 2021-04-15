open Hlo

let x = Node { op = Parameter; shape = [1000; 2]; prestine = true }
let y = Node { op = Parameter; shape = [1000; 2]; prestine = true }
let z = Node { op = Parameter; shape = [1000; 2]; prestine = true }

let xy = make_dot x y 1 1
let xyz = make_dot xy z 1 0

let () = print_endline (string_of_hlo xyz)
