"""JSON transport for ALFWorld batches; launched by empg.envs.batch."""

import contextlib
import json
import sys
import traceback


def main():
    envs = []
    transport = sys.stdout
    try:
        for line in sys.stdin:
            message = json.loads(line)
            result = None
            try:
                # Environment libraries may print; stdout belongs to the protocol.
                with contextlib.redirect_stdout(sys.stderr):
                    if message["cmd"] in ("reset", "close", "exit"):
                        for env in envs:
                            env.close()
                        envs.clear()
                    if message["cmd"] == "reset":
                        from empg.envs.alfworld_env import ALFWorldEnv
                        for task in message["tasks"]:
                            envs.append(ALFWorldEnv(task))
                        result = [env.reset() for env in envs]
                    elif message["cmd"] == "step":
                        result = [envs[i].step(a) for i, a in message["actions"]]
                response = {"result": result}
            except Exception:
                response = {"error": traceback.format_exc()}
            transport.write(json.dumps(response) + "\n")
            transport.flush()
            if message["cmd"] == "exit":
                break
    finally:
        for env in envs:
            env.close()


if __name__ == "__main__":
    main()
