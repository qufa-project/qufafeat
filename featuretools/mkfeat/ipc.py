import os

class IPC:
    def __init__(self, path):
        self.path = path

    def create(self):
        self._write_prog(0)

    def _write_prog(self, prog):
        f = open(self.path, "w")
        f.write(str(prog))
        f.close()

    def set_prog(self, prog):
        if not isinstance(prog, int):
            prog = int(prog)
        if prog >= 100:
            prog = 99
        self._write_prog(prog)

    def set_complete(self):
        self._write_prog(100)

    def get_prog(self):
        try:
            f = open(self.path, "r")
            prog = f.read()
            prog = int(prog)
            if prog == 100:
                os.remove(self.path)
            return prog
        except FileNotFoundError:
            return -1


