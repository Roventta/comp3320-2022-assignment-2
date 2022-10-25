PAPI= papi

origin_prof:
	g++ -o origin_prof ../kernel_main.cpp ../cloth_code_origin.cpp -pg -no-pie

trival_prof:
	g++ -o trival_prof ../kernel_main.cpp ../cloth_code_trival.cpp -pg

origin_papi:
	g++ -o op papi_measurement.cpp ../cloth_code.cpp -l $(PAPI)

trival_papi:
	g++ -o tp papi_measurement.cpp ../cloth_code_trival.cpp -l $(PAPI)

simd_papi:
	g++ -o sp papi_measurement.cpp ../cloth_code_sse.cpp -l $(PAPI) -mavx2 -march=native

mp_papi:
	g++ -o mp papi_measurement.cpp ../cloth_code_mp.cpp -l $(PAPI) -lgomp
clean:
	rm origin_prof
	rm trival_prof
