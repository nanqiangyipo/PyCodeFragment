class dog:
    def action_run(self):
        print(f"{id(self)} is runing dadada...")

    def action_eat(self):
        """rule"""
        print(f"{id(self)} is eating hunhun...")
        pass
    def fuli_action(self):
        print(f"{id(self)} I'm bug hiahiah..")


    def __init__(self,name):
        self.name = name