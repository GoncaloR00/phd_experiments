import json
import os

container_path = "/home/gribeiro/docker_data/containers/"

container_list = [f.name for f in os.scandir(container_path) if f.is_dir()]

for container in container_list:
    file_path = container_path + container + "/config.v2.json"
    f = open(file_path)
    data = json.load(f)
    json_string = json.dumps(data)
    # search_string = "80789e76faf29de5f12ba721e11d4c95525ada11626180813183cce0658690b1"
    # search_string = "518af036048f9e44a5677f43f93bb8aac787a247a7bdeeb633ffd49a0fd8f61e"
    # search_string = "searxng"
    search_string = "a2c381089550722c5135af2211604805ac440be1513d96fec645ad4f502e884e"
    search_string = "e8330791b36807c8f5b154e55ac1974e921c91ee143e2b772c90049cb82c8017"
    if search_string in json_string:
        print(container)

    # # string = data['NetworkSettings']['Bridge']
    # # string = str(data['State']['Running'])
    # # string = str(data['State']['Restarting'])
    # # if "80789e76faf29de5f12ba721e11d4c95525ada11626180813183cce0658690b1" in data:
    # # if "518af036048f9e44a5677f43f93bb8aac787a247a7bdeeb633ffd49a0fd8f61e" in data:
    # if container in data:
    #     print("true")
    # else:
    #     print("false")
    # # try:
    # #     string = data['NetworkSettings']['Networks']['host']['NetworkID']
        
    # #     print(container + " --> " + string)
    # # except:
    # #     print(f"Error in {container}")
    f.close()