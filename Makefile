full:
	python3 Network/train.py --instance=instances/eternity_complet.txt

trivial:
	python3 Network/train.py --instance=instances/eternity_trivial_A.txt

a:
	python3 Network/train.py --instance=instances/eternity_A.txt

b:
	python3 Network/train.py --instance=instances/eternity_B.txt

c:
	python3 Network/train.py --instance=instances/eternity_C.txt

d:
	python3 Network/train.py --instance=instances/eternity_D.txt

e:
	python3 Network/train.py --instance=instances/eternity_E.txt

trivial_game:
	python3 main.py --agent=advanced --infile=instances/eternity_trivial_A.txt


game:
	python3 main.py --agent=advanced --infile=instances/eternity_complet.txt

clean:
	rm -rf models/checkpoint/binary/*.pt
	rm -rf models/checkpoint/ordinal/*.pt
	rm -rf models/checkpoint/one_hot/*.pt


profile:
	python3 -m cProfile -s time -o out.pstats Network/train.py --instance=instances/eternity_E.txt

plot:
	gprof2dot --colour-nodes-by-selftime -f pstats out.pstats |  dot -Tpng -o output.png
