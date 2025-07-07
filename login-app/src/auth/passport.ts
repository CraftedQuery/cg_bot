import passport from 'passport';
import { Strategy as LocalStrategy } from 'passport-local';
import bcrypt from 'bcrypt';
import prisma from '../prisma';

passport.use(
  new LocalStrategy({ usernameField: 'email' }, async (email, password, done) => {
    try {
      const user = await prisma.user.findUnique({ where: { email } });
      if (!user) return done(null, false);
      const match = await bcrypt.compare(password, user.passwordHash);
      if (!match) return done(null, false);
      return done(null, user);
    } catch (err) {
      return done(err as Error);
    }
  }),
);

export default passport;
