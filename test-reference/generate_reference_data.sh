#!/bin/bash
# Generate reference data for all test configurations

echo "{"
echo '  "antialiasing": {'

# Anti-aliasing upsampling
for combo in "48000:96000:noise" "48000:96000:multitone" "44100:88200:noise" "44100:96000:noise"; do
    IFS=':' read -r in out sig <<< "$combo"
    result=$(./test_antialiasing $in $out $sig 2>/dev/null | grep "STOPBAND ATTENUATION" | awk '{print $NF}')
    echo "    \"${in}_${out}_${sig}\": ${result:-0},"
done

# Anti-aliasing downsampling
for combo in "48000:32000:alias_tones" "48000:44100:alias_tones" "96000:48000:alias_tones"; do
    IFS=':' read -r in out sig <<< "$combo"
    result=$(./test_antialiasing $in $out $sig 2>/dev/null | grep "ANTI-ALIASING ATTENUATION" | awk '{print $NF}')
    echo "    \"${in}_${out}_${sig}\": ${result:-0},"
done

echo '    "_dummy": 0'
echo '  },'

echo '  "quality": {'
# Passband ripple
for combo in "44100:48000" "48000:44100" "48000:96000" "96000:48000" "48000:32000"; do
    IFS=':' read -r in out <<< "$combo"
    ripple=$(./test_quality $in $out ripple 2>/dev/null | grep "ripple =" | awk '{print $NF}')
    max_dev=$(./test_quality $in $out ripple 2>/dev/null | grep "max_deviation =" | awk '{print $NF}')
    min_dev=$(./test_quality $in $out ripple 2>/dev/null | grep "min_deviation =" | awk '{print $NF}')
    echo "    \"ripple_${in}_${out}\": {\"ripple\": ${ripple:-0}, \"max_dev\": ${max_dev:-0}, \"min_dev\": ${min_dev:-0}},"
done

# THD at 1kHz
for combo in "44100:48000:1000" "48000:44100:1000" "48000:96000:1000" "96000:48000:1000" "48000:32000:1000" "44100:48000:10000" "48000:44100:10000"; do
    IFS=':' read -r in out freq <<< "$combo"
    thd_db=$(./test_quality $in $out thd:$freq 2>/dev/null | grep "thd_db =" | awk '{print $NF}')
    thd_pct=$(./test_quality $in $out thd:$freq 2>/dev/null | grep "thd_percent =" | awk '{print $NF}')
    echo "    \"thd_${in}_${out}_${freq}\": {\"thd_db\": ${thd_db:-0}, \"thd_percent\": ${thd_pct:-0}},"
done

# SNR
for combo in "44100:48000" "48000:44100" "48000:96000" "96000:48000" "48000:32000"; do
    IFS=':' read -r in out <<< "$combo"
    snr=$(./test_quality $in $out snr:1000 2>/dev/null | grep "snr_db =" | awk '{print $NF}')
    echo "    \"snr_${in}_${out}\": ${snr:-0},"
done

# Impulse response
for combo in "44100:48000" "48000:44100" "48000:96000" "96000:48000" "48000:32000"; do
    IFS=':' read -r in out <<< "$combo"
    pre=$(./test_quality $in $out impulse 2>/dev/null | grep "pre_ringing_db =" | awk '{print $NF}')
    post=$(./test_quality $in $out impulse 2>/dev/null | grep "post_ringing_db =" | awk '{print $NF}')
    ringout=$(./test_quality $in $out impulse 2>/dev/null | grep "ringout_samples =" | awk '{print $NF}')
    echo "    \"impulse_${in}_${out}\": {\"pre_ringing_db\": ${pre:-0}, \"post_ringing_db\": ${post:-0}, \"ringout_samples\": ${ringout:-0}},"
done

echo '    "_dummy": 0'
echo '  }'
echo "}"
