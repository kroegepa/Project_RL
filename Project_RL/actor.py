import typing
class Actor():
    def __init__ (self):
        pass
    def act(self):
        pass

class UniformBuyActor(Actor):
    def act(self, state) -> float:
        return 0.5

class AveragedBuyingActor(Actor):
    def __init__(self, amount_of_prices: int = 120, threshold: float = 0.1):
        self.current_average : float = 0
        self.price_queue = []
        #Amount of prices to be considered
        #24 prices to a day
        self.amount_of_prices = amount_of_prices
        #Deviation from average price to trigger buy sell action
        self.threshold = threshold
    def updateAverage(self, newPrice:int):
        if len(self.price_queue) == self.amount_of_prices:
            self.current_average -= self.price_queue[0]/self.amount_of_prices
            self.current_average += newPrice/self.amount_of_prices
            self.price_queue.pop(0)
            self.price_queue.append(newPrice)
        else:
            self.price_queue.append(newPrice)
            self.current_average = sum(self.price_queue)/len(self.price_queue)
    def makeDecision(self, current_price:int) -> float:
        if self.current_average == 0:
            return 1
        difference = current_price - self.current_average
        fraction  = difference/self.current_average
        #TODO maybe make sure that
        if fraction > self.threshold:
            return -1
        if fraction * -1 > self.threshold:
            return 1

        return 0
        
        

