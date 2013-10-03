import pika
import uuid
import json
from utils import log_err

class Tokens(object):
    def __init__(self):
        super(Tokens, self).__init__()
        # RabbitMQ setup
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
        self.channel = self.connection.channel()
        result = self.channel.queue_declare(exclusive=True)
        self.callback_queue = result.method.queue

        self.channel.basic_consume(self.on_response, no_ack=True,queue=self.callback_queue)

    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, msg):
        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(exchange='',
                                   routing_key='token_task_queue',
                                   properties=pika.BasicProperties(
                                         reply_to = self.callback_queue,
                                         correlation_id = self.corr_id,
                                         ),
                                   body=msg)
        while self.response is None:
            self.connection.process_data_events()
        return self.response

    def tokenize(self, s):  
        # Preprocess on the text
        s = Tokens.preprocess(s)
        

        #???? need to check connection alive
        s_map = {'orig_text':s}
        response = json.loads(self.call(json.dumps(s_map)))

        return response

    def token_length(self, s):
        s = Tokens.preprocess(s)
        s_map = {'orig_text':s}
        response = json.loads(self.call(json.dumps(s_map)))
        return response['length']

    def close_connection(self):
        self.connection.close()

    
    @staticmethod
    def preprocess(s):
        if type(s) is unicode:
            s = s.encode('ascii', 'ignore')
        else:
            s = unicode(s, errors='ignore').encode('ascii', 'ignore')

        s = s.replace("(", " ( ")
        s = s.replace(")", " ) ")
        s = s.replace(",", " , ")

        return s


if __name__ == '__main__':
    tz = Tokens()
    # tz.tokenize("A. Cau, R. Kuiper, and W.-P. de Roever. Formalising Dijkstra's development strategy within Stark's formalism. In C. B. Jones, R. C. Shaw, and T. Denvir, editors, Proc. 5th. BCS-FACS Refinement Workshop at San Francisco, 1992.")
    # tz.tokenize("O'Reilly, Cooper , G. F. (1990). The computational complexity of probabilistic inference using Bayesian belief networks. Artificial Intelligence , 42(2-3), 393-405.")
    print tz.tokenize('(5):54-89')


