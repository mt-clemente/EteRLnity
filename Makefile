full:
	python3 Network/train.py instances/eternity_complet.txt

trivial:
	python3 Network/train.py instances/eternity_trivial_A.txt

a:
	python3 Network/train.py instances/eternity_A.txt

b:
	python3 Network/train.py instances/eternity_B.txt

c:
	python3 Network/train.py instances/eternity_C.txt

d:
	python3 Network/train.py instances/eternity_D.txt

e:
	python3 Network/train.py instances/eternity_E.txt

trivial_game:
	python3 main.py --agent=advanced --infile=instances/eternity_trivial_A.txt


game:
	python3 main.py --agent=advanced --infile=instances/eternity_complet.txt