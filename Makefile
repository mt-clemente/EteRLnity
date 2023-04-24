full:
	python3 main_train.py --instance=instances/eternity_complet.txt

trivial:
	python3 main_train.py --instance=instances/eternity_trivial_A.txt

a:
	python3 main_train.py --instance=instances/eternity_A.txt

b:
	python3 main_train.py --instance=instances/eternity_B.txt

c:
	python3 main_train.py --instance=instances/eternity_C.txt

d:
	python3 main_train.py --instance=instances/eternity_D.txt

e:
	python3 main_train.py --instance=instances/eternity_E.txt

trivial_game:
	python3 main.py --agent=advanced --infile=instances/eternity_trivial_A.txt

heuristic:
	python3 main.py --agent=heuristic --infile=instances/eternity_A.txt

random:
	python3 main.py --agent=random --infile=instances/eternity_D.txt

local:
	python3 main.py --agent=local_search --infile=instances/eternity_C.txt

gameA:
	python3 main.py --agent=advanced --infile=instances/eternity_A.txt

gameB:
	python3 main.py --agent=advanced --infile=instances/eternity_B.txt

gameC:
	python3 main.py --agent=advanced --infile=instances/eternity_C.txt

gameD:
	python3 main.py --agent=advanced --infile=instances/eternity_D.txt

gameE:
	python3 main.py --agent=advanced --infile=instances/eternity_E.txt

gameF:
	python3 main.py --agent=advanced --infile=instances/eternity_complet.txt

clean:
	rm -rf models/checkpoint/*