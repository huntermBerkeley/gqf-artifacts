set(CMAKE_C_COMPILER "/opt/cray/pe/craype/2.6.2/bin/cc")
set(CMAKE_C_COMPILER_ARG1 "")
set(CMAKE_C_COMPILER_ID "Clang")
set(CMAKE_C_COMPILER_VERSION "9.0.0")
set(CMAKE_C_COMPILER_VERSION_INTERNAL "")
set(CMAKE_C_COMPILER_WRAPPER "CrayPrgEnv")
set(CMAKE_C_STANDARD_COMPUTED_DEFAULT "11")
set(CMAKE_C_COMPILE_FEATURES "c_std_90;c_function_prototypes;c_std_99;c_restrict;c_variadic_macros;c_std_11;c_static_assert")
set(CMAKE_C90_COMPILE_FEATURES "c_std_90;c_function_prototypes")
set(CMAKE_C99_COMPILE_FEATURES "c_std_99;c_restrict;c_variadic_macros")
set(CMAKE_C11_COMPILE_FEATURES "c_std_11;c_static_assert")

set(CMAKE_C_PLATFORM_ID "Linux")
set(CMAKE_C_SIMULATE_ID "")
set(CMAKE_C_SIMULATE_VERSION "")



set(CMAKE_AR "/opt/cray/pe/cce/9.1.0/binutils/x86_64/x86_64-pc-linux-gnu/bin/ar")
set(CMAKE_C_COMPILER_AR "/opt/cray/pe/cce/9.1.0/cce-clang/x86_64/bin/llvm-ar")
set(CMAKE_RANLIB "/opt/cray/pe/cce/9.1.0/binutils/x86_64/x86_64-pc-linux-gnu/bin/ranlib")
set(CMAKE_C_COMPILER_RANLIB "/opt/cray/pe/cce/9.1.0/cce-clang/x86_64/bin/llvm-ranlib")
set(CMAKE_LINKER "/opt/cray/pe/cce/9.1.0/binutils/x86_64/x86_64-pc-linux-gnu/bin/ld")
set(CMAKE_MT "")
set(CMAKE_COMPILER_IS_GNUCC )
set(CMAKE_C_COMPILER_LOADED 1)
set(CMAKE_C_COMPILER_WORKS TRUE)
set(CMAKE_C_ABI_COMPILED TRUE)
set(CMAKE_COMPILER_IS_MINGW )
set(CMAKE_COMPILER_IS_CYGWIN )
if(CMAKE_COMPILER_IS_CYGWIN)
  set(CYGWIN 1)
  set(UNIX 1)
endif()

set(CMAKE_C_COMPILER_ENV_VAR "CC")

if(CMAKE_COMPILER_IS_MINGW)
  set(MINGW 1)
endif()
set(CMAKE_C_COMPILER_ID_RUN 1)
set(CMAKE_C_SOURCE_FILE_EXTENSIONS c;m)
set(CMAKE_C_IGNORE_EXTENSIONS h;H;o;O;obj;OBJ;def;DEF;rc;RC)
set(CMAKE_C_LINKER_PREFERENCE 10)

# Save compiler ABI information.
set(CMAKE_C_SIZEOF_DATA_PTR "8")
set(CMAKE_C_COMPILER_ABI "ELF")
set(CMAKE_C_LIBRARY_ARCHITECTURE "")

if(CMAKE_C_SIZEOF_DATA_PTR)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_C_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_C_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_C_COMPILER_ABI}")
endif()

if(CMAKE_C_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "")
endif()

set(CMAKE_C_CL_SHOWINCLUDES_PREFIX "")
if(CMAKE_C_CL_SHOWINCLUDES_PREFIX)
  set(CMAKE_CL_SHOWINCLUDES_PREFIX "${CMAKE_C_CL_SHOWINCLUDES_PREFIX}")
endif()





set(CMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES "/opt/cray/pe/libsci/19.06.1/CRAYCLANG/9.0/x86_64/include;/opt/cray/pe/mpt/7.7.10/gni/mpich-crayclang/9.0/include;/opt/cray/rca/2.2.20-7.0.1.1_4.62__g8e3fb5b.ari/include;/opt/cray/alps/6.6.58-7.0.1.1_6.20__g437d88db.ari/include;/opt/cray/xpmem/2.2.20-7.0.1.1_4.21__g0475745.ari/include;/opt/cray/gni-headers/5.0.12.0-7.0.1.1_6.40__g3b1768f.ari/include;/opt/cray/dmapp/7.1.1-7.0.1.1_4.62__g38cf134.ari/include;/opt/cray/pe/pmi/5.0.14/include;/opt/cray/ugni/6.0.14.0-7.0.1.1_7.51__ge78e5b0.ari/include;/opt/cray/udreg/2.3.2-7.0.1.1_3.49__g8175d3d.ari/include;/opt/cray/wlm_detect/1.3.3-7.0.1.1_4.20__g7109084.ari/include;/opt/cray/krca/2.2.6-7.0.1.1_5.46__gb641b12.ari/include;/opt/cray-hss-devel/9.0.0/include;/usr/common/software/sles15_cgpu/boost/1.74.0/include;/opt/cray/pe/cce/9.1.0/cce-clang/x86_64/lib/clang/9.0.0/include;/opt/cray/pe/cce/9.1.0/cce/x86_64/include/craylibs;/usr/local/include;/usr/include")
set(CMAKE_C_IMPLICIT_LINK_LIBRARIES "AtpSigHandler;AtpSigHCommData;rca;sci_cray_mpi;sci_cray;pgas-dmapp;quadmath;modules;fi;craymath;f;u;csup;pthread;atomic;m;gcc;gcc_s;c;gcc;gcc_s")
set(CMAKE_C_IMPLICIT_LINK_DIRECTORIES "/opt/cray/pe/libsci/19.06.1/CRAYCLANG/9.0/x86_64/lib;/opt/cray/rca/2.2.20-7.0.1.1_4.62__g8e3fb5b.ari/lib64;/opt/cray/pe/atp/2.1.3/libApp;/opt/cray/pe/cce/9.1.0/cce/x86_64/lib;/opt/gcc/8.1.0/snos/lib/gcc/x86_64-suse-linux/8.1.0;/opt/gcc/8.1.0/snos/lib64;/lib64;/usr/lib64;/opt/gcc/8.1.0/snos/lib;/opt/cray/pe/cce/9.1.0/cce-clang/x86_64/lib;/lib;/usr/lib;/usr/common/software/sles15_cgpu/pgi/20.4/linux86-64/20.4/lib")
set(CMAKE_C_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")
