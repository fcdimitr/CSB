project('c++ CSB library', 'cpp',
        version : '1.0.0',
        default_options: ['buildtype=release'] )

csb_sources = ['csb_wrapper.cpp']

cc = meson.get_compiler('cpp')

# compiler properties and Cilk support
is_icpc      = cc.get_id().contains('intel')
use_opencilk = not is_icpc \
               and cc.has_argument('-fopencilk')
use_cilkplus = not use_opencilk \
               and (is_icpc or cc.has_argument('-fcilkplus'))

if not (use_opencilk or use_cilkplus)
  cc_str = cc.get_id() + ' ' + cc.version()
  error('The specified compiler (' + cc_str + ') does not support Cilk.')
endif

# relevant Cilk headers
cc.has_header('cilk/cilk.h'         , required : true)
cc.has_header('cilk/cilk_api.h'     , required : true)
cc.has_header('cilk/reducer_opadd.h', required : true)
cc.has_header('cilk/reducer_max.h'  , required : true)

# icpc-specific flags
if is_icpc
  add_project_arguments(['-wd3947,3946,10006,3950'], language : 'cpp')
  add_project_link_arguments(['-lirc','-limf','-lsvml'], language : 'cpp')
endif


# Cilk flags
if use_opencilk
  add_project_arguments(['-fopencilk','-DOPENCILK'], language : 'cpp')
  add_project_link_arguments(['-fopencilk'], language : 'cpp')
elif use_cilkplus
  add_project_arguments(['-fcilkplus'], language : 'cpp')
  add_project_link_arguments(['-lcilkrts'], language : 'cpp')
endif


csblib = shared_library('csb',
                        csb_sources,
                        cpp_args : ['-DALIGN=64'],
                        install : true)


pkg_mod = import('pkgconfig')
pkg_mod.generate(libraries : csblib,
                 version : '1.0',
                 name : 'libcsb',
                 filebase : 'CSB',
                 description : 'A C++ library for CSB.')
