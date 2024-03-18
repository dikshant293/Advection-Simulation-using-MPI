compile:
	mpicc -fopenmp -lm -O3 my_advection_program.c -o my_advection_program

plot:
	python -W ignore plot.py

clean:
	\rm -f *.o my_advection_program.exe my_advection_program  matrix.dat animation.dat startend.dat*~ *# 
