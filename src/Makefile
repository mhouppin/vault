#
#    Vault, a UCI-compliant chess engine derivating from Stash
#    Copyright (C) 2019-2022 Morgan Houppin
#
#    Vault is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Vault is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

SOURCES := $(wildcard sources/*.c)

OBJECTS := $(SOURCES:%.c=%.o)
DEPENDS := $(SOURCES:%.c=%.d)

ARCH ?=
NATIVE ?= yes
CFLAGS ?= -O3 -flto -DNDEBUG
CPPFLAGS ?= -Werror
LDFLAGS ?=

user_CFLAGS := $(CFLAGS)
user_CPPFLAGS := $(CPPFLAGS)
user_LDFLAGS := $(LDFLAGS)

own_CFLAGS := -std=gnu11
own_CPPFLAGS := -Wall -Wextra -Wcast-qual -Wshadow -Wvla -MMD -I include
own_LDFLAGS := -lpthread -lm

# If no arch is specified, we select a prefetch+popcnt build by default.
# If you want to build on a different architecture, please specify it in the
# command with ARCH=something. If you want to disable all arch-specific
# optimizations, use ARCH=unknown.

ifeq ($(ARCH),)
	ARCH=x86-64-popcnt
endif

# Add .exe to the executable name if we are on Windows

ifeq ($(OS),Windows_NT)
	EXE = vault.exe
	own_LDFLAGS += -static
else
	EXE = vault
endif

ifeq ($(findstring -32,$(ARCH)),-32)
	ARCH_CFLAGS += -m32
else
	ARCH_CFLAGS += -m64
endif

ifeq ($(findstring x86-32,$(ARCH)),x86-32)
	ARCH_CFLAGS += -mmmx
endif

ifeq ($(findstring x86-64,$(ARCH)),x86-64)
	ARCH_CFLAGS += -msse -msse2
endif

ifeq ($(findstring -sse,$(ARCH)),-sse)
	ARCH_CFLAGS += -msse
endif

ifeq ($(findstring -popcnt,$(ARCH)),-popcnt)
	ARCH_CFLAGS += -msse3 -mpopcnt
endif

ifeq ($(findstring -mmx,$(ARCH)),-mmx)
	ARCH_CFLAGS += -mmmx
endif

ifeq ($(findstring -sse2,$(ARCH)),-sse2)
	ARCH_CFLAGS += -msse -msse2
endif

ifeq ($(findstring -ssse3,$(ARCH)),-ssse3)
	ARCH_CFLAGS += -msse -msse2 -mssse3
endif

ifeq ($(findstring -sse41,$(ARCH)),-sse41)
	ARCH_CFLAGS += -msse -msse2 -mssse3 -msse4.1
endif

ifeq ($(findstring -modern,$(ARCH)),-modern)
	ARCH_CFLAGS += -mpopcnt -msse -msse2 -msse3 -mssse3 -msse4.1
endif

ifeq ($(findstring -avx2,$(ARCH)),-avx2)
	ARCH_CFLAGS += -mpopcnt -msse -msse2 -msse3 -mssse3 -msse4.1 -mavx2
endif

ifeq ($(findstring -bmi2,$(ARCH)),-bmi2)
	own_CFLAGS += -DUSE_PEXT
	ARCH_CFLAGS += -mpopcnt -msse -msse2 -msse3 -mssse3 -msse4.1 -mavx2 -mbmi2
endif

ifeq ($(findstring -avx512,$(ARCH)),-avx512)
	own_CFLAGS += -DUSE_PEXT
	ARCH_CFLAGS += -mpopcnt -msse -msse2 -msse3 -mssse3 -msse4.1 -mavx2 -mavx512f -mavx512bw -mbmi2
endif

# If native is specified, build will try to use all available CPU instructions

ifeq ($(NATIVE),yes)
    own_CFLAGS += -march=native
else
	own_CFLAGS += $(ARCH_CFLAGS)
endif

override CFLAGS += $(own_CFLAGS)
override CPPFLAGS += $(own_CPPFLAGS)
override LDFLAGS += $(own_LDFLAGS)

all: $(EXE)

$(EXE): $(OBJECTS)
	+$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo ''
	@echo '    Short notice: native builds are enabled by default on the project.'
	@echo '    To build more portable binaries, please add `NATIVE=no` to your'
	@echo '    make command (like `make ARCH=x86-64-modern NATIVE=no` for example).'
	@echo ''

-include $(DEPENDS)

clean:
	rm -f $(OBJECTS) $(DEPENDS)

fclean: clean
	rm -f $(EXE)

re:
	$(MAKE) fclean
	+$(MAKE) all CFLAGS="$(user_CFLAGS)" CPPFLAGS="$(user_CPPFLAGS)" LDFLAGS="$(user_LDFLAGS)"

.PHONY: all clean fclean re
