# Poner a 0 la variable total
empezados=0
total=0
completados=0
for file in $(ls ./csvs/); do
    # Sumar a total el numero de lineas de cada archivo
    #wc -l ./csvs/$file
    total=$(($total + $(wc -l ./csvs/$file | cut -d " " -f 1)-1))
    # Si el número de lineas de el archivo es exactamente 501, es que se ha completado. Aumentar el contador de completados.
    if [ $(wc -l ./csvs/$file | cut -d " " -f 1) -eq 501 ]; then
        completados=$(($completados + 1))
    else
        echo "Archivo: $file, lineas: $(wc -l ./csvs/$file | cut -d " " -f 1)"
    fi
    empezados=$(($empezados + 1))
        
done

# elements tiene que ser 501 veces el número de archivos que haya en ./csvs/ 
all_Experiments=$((8*4*7*9))
elements=$(( $all_Experiments * 500))
percent=$(($total * 100 / $elements))
echo "Total: $total/$elements = ${percent}%"
percent=$(($completados * 100 / $all_Experiments))
echo "Completados: $completados/$all_Experiments = ${percent}%"

echo "Empezados: $empezados/$all_Experiments = $(($empezados * 100 / $all_Experiments))%"

