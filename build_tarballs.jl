# Note that this script can accept some limited command-line arguments, run
 # `julia build_tarballs.jl --help` to see a usage message.
 using BinaryBuilder, Pkg

 name = "CSB"
 version = v"1.0.0"

 # Collection of sources required to complete build
 sources = [
     GitSource("https://github.com/fcdimitr/CSB.git", "7a03c8fb85a85759f69a782b07c99d13c0d80d5a")
 ]

 # Bash recipe for building across all platforms
 script = raw"""
 cd $WORKSPACE/srcdir/csb/
 meson --cross-file=${MESON_TARGET_TOOLCHAIN%.*}_gcc.meson build
 cd build/
 ninja -j${nproc}
 ninja install
 """

 # These are the platforms we will build for by default, unless further
 # platforms are passed in on the command line
  platforms = [
    Platform("i686", "linux"; libc = "glibc"),
    Platform("x86_64", "linux"; libc = "glibc"),
    Platform("i686", "linux"; libc = "musl"),
    Platform("x86_64", "linux"; libc = "musl"),
    Platform("x86_64", "macos"; ),
    Platform("x86_64", "freebsd"; )
  ]


 # The products that we will ensure are always built
 products = [
     LibraryProduct("libcsb", :libcsb)
 ]

 # Dependencies that must be installed before this package can be built
 dependencies = [
     Dependency(PackageSpec(name="cilkrts_jll", uuid="71772805-00bc-5a29-9044-a26d819b7806"))
 ]

 # Build the tarballs, and possibly a `build.jl` as well.
 build_tarballs(ARGS, name, version, sources, script, platforms, products, dependencies; julia_compat="1.6", preferred_gcc_version = v"7.1.0")
