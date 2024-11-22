#Compiler commands
CC=gcc
MPICC=mpicc

#enable/disable
ENABLE_MAGICK_WAND=true
ENABLE_COLORS=true
ENABLE_AUTO_CORRECTION=true

#Other system commands
RM=rm -f
RMDIR=rmdir
CP=cp
MAKEDEPEND=makedepend
MKDIR=mkdir
TAR=tar

#flags
CFLAGS=-Wall -g -O3 -march=native
LDFLAGS=-lm

#switch to correction automatically
ifeq ($(ENABLE_AUTO_CORRECTION),true)
	MODE=$(shell test -e exercise_1_correction.c && echo '_correction')
else
	MODE=
endif

#Files
LBM_LIB_SOURCES=src/lbm_phys.c \
                src/lbm_init.c \
                src/lbm_struct.c \
                src/lbm_comm.c \
                src/lbm_config.c \
                src/lbm_save.c \
                exercise_0.c \
                exercise_6$(MODE).c \
                src/exercises.c

#Compute paths
LBM_LIB_OBJECTS=$(addprefix objs/, $(LBM_LIB_SOURCES:.c=.o))
LBM_SOURCES=src/main.c $(LBM_LIB_SOURCES)
LBM_HEADERS=$(wildcards:*.h)
LBM_OBJECTS=$(addprefix objs/, $(LBM_SOURCES:.c=.o))
LBM_ARCHIVE=lbm_sources.tar.bz2

#Targets
TARGET=lbm display check_comm

#MagickWand for image
ifeq ($(ENABLE_MAGICK_WAND),true)
	MAGICK_WAND_CFLAGS=$(shell pkg-config MagickWand --cflags) -DHAVE_MAGICK_WAND
	MAGICK_WAND_LDFLAGS=$(shell pkg-config MagickWand --libs)
	CFLAGS+=$(MAGICK_WAND_CFLAGS)
	LDFLAGS+=$(MAGICK_WAND_LDFLAGS)
endif

#disable colors
ifneq ($(ENABLE_COLORS),true)
	CFLAGS+=-DDISABLE_COLORS
endif

#Default rule
all: objs $(TARGET)

#Create object dir
objs:
	$(MKDIR) -p objs
	$(MKDIR) -p objs/src

#Compile .c => .o
objs/%.o: %.c
	$(MPICC) $(CFLAGS) -c -o $@ $<

#Alias to create archive
archive: $(LBM_ARCHIVE)

# Create archive
$(LBM_ARCHIVE): *.c src cases Makefile config.txt *.sh *.md
	$(RM) -rfvd lbm_sources
	$(MKDIR) lbm_sources
	$(CP) -r $^ lbm_sources
	$(TAR) -cvjf $(LBM_ARCHIVE) lbm_sources
	$(RM) -rd lbm_sources

# Build main executable
lbm: $(LBM_OBJECTS)
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Build displayer
display: src/display.c
	$(CC) $(CFLAGS) -o $@ $<

# Build comm checker
check_comm: src/check_comm.c $(LBM_LIB_OBJECTS)
	$(MPICC) $(CFLAGS) -o $@ $< $(LBM_LIB_OBJECTS) $(LDFLAGS)

# Clean
clean:
	$(RM) $(LBM_OBJECTS)
	$(RM) $(TARGET)
	$(RM) $(LBM_ARCHIVE)
	$(RM) -rd objs
	$(RM) *~
	$(RM) *.swp
	$(RM) .*.swp
	$(RM) */*.swp
	$(RM) */.*.swp
	$(RM) -r __python__

# Gen deps
depend:
	$(MAKEDEPEND) -Y. $(LBM_SOURCES) src/display.c src/check_comm.c

#Tasks to always run
.PHONY: clean all depend archive

# DO NOT DELETE

objs/src/display.o: src/lbm_struct.h src/lbm_config.h
objs/src/check_comm.o: src/lbm_comm.h src/lbm_struct.h src/lbm_config.h
objs/src/main.o: src/lbm_config.h src/lbm_struct.h src/lbm_phys.h src/lbm_comm.h src/lbm_init.h src/lbm_save.h
objs/src/main.o: src/exercises.h
objs/src/lbm_phys.o: src/lbm_config.h src/lbm_struct.h src/lbm_phys.h src/lbm_comm.h
objs/src/lbm_init.o: src/lbm_phys.h src/lbm_struct.h src/lbm_config.h src/lbm_comm.h src/lbm_init.h
objs/src/lbm_struct.o: src/lbm_struct.h src/lbm_config.h
objs/src/lbm_comm.o: src/lbm_comm.h src/lbm_struct.h src/lbm_config.h
objs/src/lbm_config.o: src/lbm_config.h
objs/src/lbm_save.o: src/lbm_phys.h src/lbm_struct.h src/lbm_config.h src/lbm_comm.h src/lbm_save.h
objs/exercise_0.o: src/lbm_struct.h src/lbm_config.h src/exercises.h src/lbm_comm.h src/lbm_save.h src/lbm_phys.h
objs/exercise_6$(MODE).o: src/lbm_struct.h src/lbm_config.h src/exercises.h src/lbm_comm.h src/lbm_save.h src/lbm_phys.h
oobjs/exercises.o: src/exercises.h src/lbm_comm.h src/lbm_struct.h src/lbm_config.h src/lbm_save.h src/lbm_phys.h
objs/display.o: src/lbm_struct.h src/lbm_config.h
