hs apply -f serving.yaml
python3 demo/simulate_traffic.py --cluster
echo "Test is ready"
