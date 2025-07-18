# for timeout in '0.810' '1.620'; do
#     python main.py --SARSOP --env_name=sunysb --filename="sunysb_sarsop_timeout${timeout}.txt" --timeout=${timeout}
# done

# for timeout in '3.920'; do
#     python main.py --SARSOP --env_name=fourth --filename="fourth_sarsop_timeout${timeout}.txt" --timeout=${timeout}
# done

# for timeout in '0.720' '1.440' '2.880'; do
#     python main.py --SARSOP --env_name=TagAvoid --filename="TagAvoid_sarsop_timeout'${timeout}'.txt" --timeout=${timeout}
# done

for timeout in '5.025' '10.050'; do
    python main.py --SARSOP --env_name=under_water --filename="under_water_sarsop_timeout${timeout}.txt" --timeout=${timeout}
done
