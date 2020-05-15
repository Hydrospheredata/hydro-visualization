hs apply -f serving.yaml
python3 demo/simulate_traffic.py --cluster localhost:9090
echo "Test is ready"
