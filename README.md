pip install git-filter-repo

git clone https://github.com/ngthienvu/crypto-bot-telegram.git
cd crypto-bot-telegram

git filter-repo --invert-paths --path .env --force

echo ".env" >> .gitignore
git add .
git commit -m "Cleaned sensitive data"
git push origin --force
