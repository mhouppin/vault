set -e

version=0.4

cd src

for OS in windows linux
do
    for arch in \
        general-32 \
        general-64 \
        x86-32 \
        x86-32-sse2 \
        x86-32-sse41-popcnt \
        x86-64 \
        x86-64-sse3-popcnt \
        x86-64-ssse3 \
        x86-64-modern \
        x86-64-avx2 \
        x86-64-bmi2 \
        x86-64-avx512
    do
        compiler=gcc
        extra_ldflags=
        barch="${arch/x86-64/x86_64}"
        binary_name="vault-$version-$OS-$barch"

        if [ $OS == windows ]
        then
            case "$arch" in
                *-32*) compiler=i686-w64-mingw32-gcc;;
                *) compiler=x86_64-w64-mingw32-gcc;;
            esac
            binary_name+=.exe
            extra_ldflags=-static
        fi

        make fclean
        LDFLAGS="$extra_ldflags" CC="$compiler" make EXE="$binary_name" ARCH="$arch" native=no
    done
done
