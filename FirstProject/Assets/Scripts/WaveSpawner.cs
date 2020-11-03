using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;
using TMPro;


public class WaveSpawner : MonoBehaviour
{
    public enum SpawnState { SPAWNING, WAITING, COUNTING };

    [System.Serializable]
    public class Wave
    {
        public string name;
        public int count;
        public float rate;
    }

    public Wave[] waves;
    public int nextWave = 0;
    public float timeBetweenWaves = 5f;
    public float waveCountdown;

    public TextMeshProUGUI text;
    public GameObject canvas;

    private float searchCountdown = 1f;

    private SpawnState state = SpawnState.COUNTING;


    public GameObject mainMarker;
    public GameObject enemy1;

    public int minimum_range = 10;
    public int maximum_range = 15;

    private static LinkedList<GameObject> enemies;

    private bool spawning = false;

    public void startSpawning()
    {
        spawning = true;
    }

    // Start is called before the first frame update
    void Start()
    {
        StartCoroutine(printMessage("Starting wave spawner!"));
        waveCountdown = timeBetweenWaves;
        enemies = new LinkedList<GameObject>();
    }

    private IEnumerator printMessage(string str){
        text.text = str;
        canvas.SetActive(true);
        yield return new WaitForSeconds(3);
        canvas.SetActive(false);
    }

    // Update is called once per frame
    void Update()
    {
        if (spawning)
        {
            if (state == SpawnState.WAITING)
            {
                if (!EnemyIsAlive())
                {
                    StartCoroutine(printMessage("Wave Completed"));
                    WaveCompleted();
                }
                else
                {
                    return;
                }
            }

            if (waveCountdown <= 0)
            {
                if (state != SpawnState.SPAWNING)
                {
                    // Start spawning wave
                    StartCoroutine(SpawnWave(waves[nextWave]));
                }
            }
            else
            {
                waveCountdown -= Time.deltaTime;
            }
        }
    }

    private void WaveCompleted()
    {

        state = SpawnState.COUNTING;
        waveCountdown = timeBetweenWaves;

        if (nextWave + 1 > waves.Length - 1)
        {
            nextWave = 0;
            StartCoroutine(printMessage("ALL WAVES COMPLETE! Increasing level..."));
        

            for(int i = 0; i < waves.Length; i++)
            {
                waves[i].count += 10;
                var navMeshComp = enemy1.GetComponent(typeof(NavMeshAgent)) as NavMeshAgent;

                if(navMeshComp.speed < 4)
                {
                    navMeshComp.speed += 0.3f;
                }
            }

        }
        else
        {

            GameObject[] turrets;
            turrets = GameObject.FindGameObjectsWithTag("Turret");
            foreach(GameObject turret in turrets)
            {
                turret.GetComponent<FindAndFireTower>().replenishBullets();
            }

            nextWave++;
        }
    }

    private bool EnemyIsAlive()
    {
        searchCountdown -= Time.deltaTime;
        if (searchCountdown <= 0f)
        {
            searchCountdown = 1f;
            if (GameObject.FindGameObjectWithTag("Enemy") == null)
            {
                return false;
            }
        }
        return true;

    }

    private IEnumerator SpawnWave(Wave _wave)
    {
        StartCoroutine(printMessage("Spawning wave: " + _wave.name));
   
        state = SpawnState.SPAWNING;

        // Spawn
        for (int i = 0; i < _wave.count; i++)
        {
            SpawnEnemy();
            yield return new WaitForSeconds(1f / _wave.rate);
        }

        state = SpawnState.WAITING;

        yield break;
    }

    private void SpawnEnemy()
    {
        Debug.Log("Spawning enemy: " + enemy1.name);

        // Spawn enemy
        Vector2 circleRandom = Random.insideUnitCircle.normalized * Random.Range(10, 15);
        Vector3 pos = new Vector3(circleRandom.x, 0f, circleRandom.y);
        var obj = Instantiate(enemy1, pos, Quaternion.identity);
        obj.transform.SetParent(mainMarker.transform);
        //Debug.Log(obj);
        enemies.AddFirst(obj);
    }

    public static void deleteEnemyFromWave(GameObject enemy)
    {
        enemies.Remove(enemy);
    }
    public static LinkedList<GameObject> getEnemiesFromWave()
    {
        return enemies;
    }
}
