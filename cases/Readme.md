Running advanced cases
======================

The lbm executable can support loading images to define the obstacle to simulate instead
of using a simple circle. You can define the obstacle in black in a PNG image with a white
background and load it into LBM.

CAUTION
-------

Those cases are configured to get a nice output, it never been made to be
realistic as configuration such as the Reynold number are choosed 'randomly' to
get nice effect.

Compiling ImageMagick support
-----------------------------

In order to use images you need to compile LBM with the ImageMagick suppport. Open
the `Makefile` and set the `ENABLE_MAGICK_WAND` to `true`.

```Makefile
#enable/disable
ENABLE_MAGICK_WAND=true
```

You will need to install MagickWand to procceed, under debian this is the
`libmagickwand-dev` package.

Then recompile:

```sh
# clean before recompiling
make clean

# recompile
make
```

Running
-------

Run from the source directory:

```sh
# complex case
mpirun -np 8 ./lbm -c cases/config-complex.txt
./gen_animate_gif.sh output.raw output-complex.gif

# truck case
mpirun -np 8 ./lbm -c cases/config-track.txt
./gen_animate_gif.sh output.raw output-truck.gif

# wing case
mpirun -np 8 ./lbm -c cases/config-wing.txt
./gen_animate_gif.sh output.raw output-wing.gif
```
