import { Router } from 'express';
import { requireAuth } from '../middleware/requireAuth';

const router = Router();

router.get('/dashboard-data', requireAuth, (req, res) => {
  res.json({ message: 'Protected content' });
});

export default router;
