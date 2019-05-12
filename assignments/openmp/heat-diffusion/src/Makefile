CC = gcc
CFLAGS = -std=c99 -O3 -Wall -fopenmp

LIBS = -lm

ODIR = obj

_OBJ = trap.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: %.c
	$(CC) -c -o $@ $< $(CFLAGS)

trap.exe: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core *.exe
