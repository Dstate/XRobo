import json_numpy
import requests

class Deploy_Client:
    def __init__(self, host="0.0.0.0", port="8000"):
        self.host = host
        self.port = port
        self.url = f"http://{host}:{port}/act" 
        self.info_url = f"http://{host}:{port}/info" 
    
    def get_action(self, agent_view_images, wrist_view_images, proprio, instruction, eval_horizon, t=-1):
        query = {
            "agent_view_images": json_numpy.dumps(agent_view_images),
            "wrist_view_images": json_numpy.dumps(wrist_view_images),
            "proprio": json_numpy.dumps(proprio),
            "instruction": instruction,
            "eval_horizon": eval_horizon,
            "t" : t
        }
        req = requests.post(self.url, json=query).json()

        if "error_str" in req.keys():
            print("Error occurred:", req["error_str"])
            return None
        else:
            return req['pred_action']
    
    def info_query(self):
        req = requests.post(self.info_url).json()
        return req['save_path']
    

if __name__ == '__main__':
    client = Deploy_Client(host="0.0.0.0", port=8989+7)
    print(client.info_query())