full:
	python3 Network/train.py instances/eternity_complet.txt

trivial:
	python3 Network/train.py instances/eternity_trivial_A.txt

trivial_game:
	python3 main.py --agent=advanced --infile=instances/eternity_trivial_A.txt


game:
	python3 main.py --agent=advanced --infile=instances/eternity_complet.txt