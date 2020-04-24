hs apply -f serving.yaml
python3 demo/simulate_traffic.py --cluster localhost
echo "Test is ready"
