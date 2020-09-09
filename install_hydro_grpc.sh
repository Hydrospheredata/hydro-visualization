pip uninstall hydro_serving_grpc -y
rm -rf hydro-serving-protos
git clone https://github.com/Hydrospheredata/hydro-serving-protos --branch chore/vis-grpc-endpoint
cd hydro-serving-protos
pip install --user -r python-package/requirements.txt
make python
cd  python-package
python setup.py install --user
cd ../..
