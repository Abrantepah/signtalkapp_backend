import json
from channels.generic.websocket import AsyncWebsocketConsumer
from urllib.parse import parse_qs


class ConsultationConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.session_id = self.scope['url_route']['kwargs']['session_id']
        self.room_group_name = f"consult_{self.session_id}"

        # Parse the query string safely
        query_string = self.scope.get('query_string', b'').decode()
        params = parse_qs(query_string)
        
        # Get the role (it returns a list, so we take the first item)
        user_role = params.get('role', [None])[0]

        await self.channel_layer.group_add(self.room_group_name, self.channel_name)
        await self.accept()

        print(f"Connection attempt by: {user_role} in room {self.session_id}")

        # ONLY notify the room if the joiner is the patient
        if user_role == 'patient':
            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    'type': 'patient_joined',
                    'message': {'status': 'connected'}
                }
            )
            
    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.room_group_name, self.channel_name)

    async def receive(self, text_data):
        data = json.loads(text_data)
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'broadcast_message',
                'message': data
            }
        )

    async def broadcast_message(self, event):
        # Format for Flutter broadcast listener
        await self.send(text_data=json.dumps({
            'type': 'broadcast_message',
            'message': event['message']
        }))
        
    async def patient_joined(self, event):
        # Format for Flutter modal listener
        await self.send(text_data=json.dumps({
            'type': 'patient_joined',
            'message': event['message']
        }))