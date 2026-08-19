1. 설치

pip install -r requirements.txt

2. offline dataset 생성

python offline_dataset_generate.py

3. 학습

python main.py --model hiql

4. 평가

python hiql_evaluation.py   --checkpoint checkpoints/hiql_0804/hiql_10000.msgpack   --episodes 300   --max-episode-steps 50000 --eval-temperature 0.5 --log-interval 1000

python hiql_run_pretrained.py  --checkpoint checkpoints/hiql_0804/hiql_10000.msgpack --eval-temperature 0.5


그외.

포켓몬 게임 즐기기

pyboy red_rom/PokemonRed.gb



*참고*
100% GPT 작성이라 틀린 부분이 있을 수 있음. 논문 읽으면서 코드 작성/확인해보아요.
