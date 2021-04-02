#!/bin/sh

#$ -S /bin/sh
#$ -cwd
#$ -V
#$ -q all.q@ichigo
#$ -q all.q@yuzu
#$ -q all.q@suika
#$ -q all.q@mikan
#$ -q all.q@kaki
#$ -M k_eguchi@r.phys.nagoya-u.ac.jp
#$ -m a

./a.out
