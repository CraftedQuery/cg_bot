import bcrypt from 'bcrypt';
import prisma from './prisma';

async function main() {
  const passwordHash = await bcrypt.hash('password', 12);
  await prisma.user.create({
    data: { email: 'user@example.com', passwordHash, provider: 'local' },
  });
  console.log('Seed user created');
}

main()
  .catch((e) => console.error(e))
  .finally(async () => {
    await prisma.$disconnect();
  });
