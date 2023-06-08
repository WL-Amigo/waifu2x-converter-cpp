gen() {
    mode=$1
    bindir=$2

    echo $mode
    echo $bindir

    dst_dir=`date '+%m%d/waifu2x-converter_'$mode'_%m%d'`

    /bin/rm -rf $dst_dir

    mkdir -p $dst_dir

    SYSNAME=`uname -s`
    if test $SYSNAME = 'Linux'
    then
        EXE=
    else
        EXE=.exe

        cp $bindir/w2xc.dll $dst_dir
        cp w32-apps/install.js $dst_dir
        cp w32-apps/install.bat $dst_dir
        cp w32-apps/uninstall.js $dst_dir
        cp w32-apps/uninstall.bat $dst_dir
        cp $bindir/w2xcr.exe $dst_dir
        cp dists/README.jp $dst_dir
        cp dists/README.en $dst_dir

        mkdir $dst_dir/libw2xc
        mkdir $dst_dir/libw2xc/samples
        cp $bindir/w2xc.lib $dst_dir/libw2xc
        cp src/w2xconv.h $dst_dir/libw2xc
        cp w32-apps/w2xc.c $dst_dir/libw2xc/samples
        cp w32-apps/w2xcr.c $dst_dir/libw2xc/samples
        cp w32-apps/w2xcr.h $dst_dir/libw2xc/samples
        cp w32-apps/w2xcr.rc $dst_dir/libw2xc/samples
        cp w32-apps/Makefile $dst_dir/libw2xc/samples

        mkdir $dst_dir/ExtendedSendTo
        cp w32-apps/ExtendedSendTo/* $dst_dir/ExtendedSendTo
    fi

    cp $bindir/waifu2x-converter-cpp$EXE $dst_dir/waifu2x-converter_$mode$EXE
    cp LICENSE $dst_dir

    mkdir $dst_dir/models
    cp models/*.json $dst_dir/models

    mkdir $dst_dir/models_rgb
    cp models_rgb/*.json $dst_dir/models_rgb
}



x86_bindir=$1
x64_bindir=$2
if test "$x86_bindir" = "" -o "$x64_bindir" = ""
then
    echo "usage dist.sh <x86dir> <x64dir>"
    exit 1
fi

gen x86 $x86_bindir
gen x64 $x64_bindir