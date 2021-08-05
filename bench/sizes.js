const moduleTxt = await Deno.readTextFile(Deno.args[0]);

const bytesPerElem = 64 / 8;

const maxConfig = 7 * 1000000000;

for (const [match, shape] of moduleTxt.matchAll(/f64\[([^\]]+)\]/ig)) {
  const dims = shape.split(",").map(n => parseInt(n));
  const size = dims.reduce((acc, dim) => acc * dim, bytesPerElem);
  if (size > maxConfig)
    console.log(match, dims, size);
}
