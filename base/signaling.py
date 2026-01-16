import json
from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async
from .models import ConsultationSession


class ConsultationConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.session_id = self.scope['url_route']['kwargs']['session_id']
        self.room_group_name = f"consult_{self.session_id}"

        session = await sync_to_async(ConsultationSession.objects.get)(
            id=self.session_id
        )

        if not session.is_active or session.is_expired():
            await self.close()
            return

        await self.channel_layer.group_add(self.room_group_name, self.channel_name)
        await self.accept()
    async def receive(self, text_data):
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'broadcast_message',
                'message': json.loads(text_data)
            }
        )
    async def broadcast_message(self, event):
        await self.send(text_data=json.dumps(event['message']))
