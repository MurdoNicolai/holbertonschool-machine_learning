-- lists all bands with Glam rock as their main style,
SELECT band_name,
       CASE
           WHEN split IS NULL THEN YEAR(CURDATE()) - formed
           ELSE split - formed
       END AS lifespan_years
FROM metal_bands
WHERE FIND_IN_SET('Glam rock', style) > 0
ORDER BY lifespan_years DESC;
