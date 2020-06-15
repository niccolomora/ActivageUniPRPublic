cd ../out/figures/
echo "Copying template..."
cp legenda/*.png .
echo "Done."
echo "Merging together images..."
convert *.png SensorsReport.$(date +"%Y-%m-%d").pdf
echo "Done."
echo "Cleaning up files..."
rm *.png
echo "Done."
