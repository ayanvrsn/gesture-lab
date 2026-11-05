import io,unittest
from PIL import Image
from app import app
from prepare import DATA
class DemoTests(unittest.TestCase):
    def setUp(self): self.client=app.test_client()
    def test_routes(self):
        self.assertEqual(self.client.get('/').status_code,200)
        self.assertEqual(self.client.get('/api/info').json['device'],'CPU')
    def test_bad_uploads(self):
        self.assertEqual(self.client.post('/api/predict').status_code,400)
        self.assertEqual(self.client.post('/api/predict',data={'image':(io.BytesIO(b'bad'),'bad.jpg')}).status_code,400)
        self.assertEqual(self.client.post('/api/predict',data={'image':(io.BytesIO(b'x'*(11*1024*1024)),'large.jpg')}).status_code,413)
    def test_prediction_and_stress(self):
        p=next((DATA/'test/peace').glob('*.jpg'))
        for kind in ['original','dark','blur','noise','flip','rotate','bright']:
            with p.open('rb') as f:r=self.client.post('/api/predict',data={'image':(f,'hand.jpg'),'transform':kind})
            self.assertEqual(r.status_code,200);d=r.json
            self.assertAlmostEqual(sum(d['probabilities'].values()),1,places=5)
            self.assertTrue(d['heatmap'].startswith('data:image/jpeg;base64,'))
            self.assertGreater(d['inference_ms'],0)
    def test_live_payload_and_explanation(self):
        p=next((DATA/'test/peace').glob('*.jpg'))
        with p.open('rb') as f:
            live=self.client.post('/api/predict',data={'image':(f,'frame.jpg'),'explain':'0','include_images':'0'})
        self.assertEqual(live.status_code,200)
        self.assertFalse(live.json['explained'])
        self.assertNotIn('input',live.json)
        self.assertNotIn('heatmap',live.json)
        with p.open('rb') as f:
            explained=self.client.post('/api/predict',data={'image':(f,'frame.jpg'),'explain':'1','include_images':'1'})
        self.assertTrue(explained.json['explained'])
        self.assertIn('heatmap',explained.json)
        self.assertEqual(live.json['label'],explained.json['label'])
        for key,value in live.json['probabilities'].items():
            self.assertAlmostEqual(value,explained.json['probabilities'][key],places=5)
if __name__=='__main__':unittest.main()
