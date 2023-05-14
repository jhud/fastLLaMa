"""
WebSocket server to serve up responses to prompts given in JSON.
An example:
command = {"command": "start", "model": "alpaca", "mode": "by_sentence", "input": "How do I pat a dog?"}
{"command": "reload_model"} = reset the model.
"""
import asyncio
import json
import sys
import websockets

sys.path.append("./build/")

import fastLlama

# Wherever your models are on disk
MODEL_ID = "ALPACA-LORA-7B"
#MODEL_PATH = "../llm_models/alpaca/13B/ggml-model-q4_0.bin"
MODEL_PATH = "../llm_models/alpaca/7B/ggml-model-q4_0.bin"


#res = model.save_state("./saves/fast_llama.bin") #save model state

#res = model.load_state("./saves/fast_llama.bin") #load model state
#if not res:
#    print("\nFailed to load the model")
#    exit(1)
#print("\nLoaded the model successfully!")



import threading
import websocket
import time

class LLMServer:

    def __init__(self, host, port):

        self.host = host
        self.port = port

        self.load_model()

    def start(self):
        self.server.run_forever()

    def load_model(self):
        print("Loading model...")
        self.model = fastLlama.Model(
            id=MODEL_ID,
            path=MODEL_PATH,  # path to model
            num_threads=8,  # number of threads to use
            n_ctx=512,  # context size of model
            last_n_size=64,  # size of last n tokens (used for repetition penalty) (Optional)
            seed=0  # seed for random number generator (Optional)
        )
        print("Loaded!\n")


    async def handle_command(self, text, ws):
        message = json.loads(text)
        command = message["command"]
        model = message.get("model", "generic")

        if command == "reload_model":
            self.load_model()
        elif command == "start":
            out_queue = []
            input = message["input"]
            mode = message.get("mode", "none")

            prompt = input

            if model == "alpaca":
                prompt = f"### Instruction:\n\n{input}\n\n ### Response:\n\n"

            print("Prompt: " + prompt + "\n")

            t = threading.Thread(target=self.process_prompt_blocking, args=(prompt, out_queue))
            t.start()

            if mode == "atomic":
                while t.is_alive():
                    await asyncio.sleep(0.1)
                await ws.send("".join(out_queue))
            elif mode == "sentence":
                while t.is_alive():
                    await asyncio.sleep(0.1)
                    for i, token in enumerate(out_queue):
                        if "." in token:
                            to_send = out_queue[:i+1]
                            await ws.send("".join(to_send))
                            for v in range(0, i+1):
                                out_queue.pop(0)
                            break
                await ws.send("".join(out_queue))
            else:
                while t.is_alive():
                    await asyncio.sleep(0.1)
                    while len(out_queue) > 0:
                        token = out_queue.pop(0)
                        await ws.send(token)

        elif command == "stop":
            # TODO: implement stopping the long running function
            pass
        else:
            await ws.send(f"Unknown command: {command}")

    def process_prompt_blocking(self, prompt: str, out_queue: list):

        def stream_token(x: str) -> None:
            """
            This function is called by the llama library to stream tokens
            """
            print(x, end='', flush=True)
            out_queue.append(x)

        res = self.model.ingest(prompt)  # ingest model with prompt

        if res != True:
            print("\nFailed to ingest model")
            exit(1)

        res = self.model.generate(
            num_tokens=50,
            top_p=0.95,  # top p sampling (Optional)
            temp=0.8,  # temperature (Optional)
            repeat_penalty=1.3,  # repetition penalty (Optional)
            streaming_fn=stream_token,  # streaming function
            stop_word=[".\n", "#"]  # stop generation when this word is encountered (Optional)
        )

        print("Finished generation\n")

    async def on_connection(self, ws, path):
        print("New connection established")

        try:
            async for message in ws:
                print(f"MSG: {message}\n")
                await self.handle_command(message, ws)
        except websockets.exceptions.ConnectionClosed:
            print("Connection closed")

    async def run(self):
        self.server = await websockets.serve(self.on_connection, self.host, self.port)
        await asyncio.Future()

if __name__ == "__main__":
    async def main():
        server = LLMServer("localhost", 3001)
        await server.run()

    asyncio.run(main())