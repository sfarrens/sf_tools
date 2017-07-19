rm -rf *.html _sources _modules _static
git checkout master sf_tools/docs/build/html
mv sf_tools/docs/build/html/* .
rm -r docs
git add .
git commit -m "updated html"
git push origin gh-pages
