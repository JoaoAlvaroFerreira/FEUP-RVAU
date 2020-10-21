using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GameController : MonoBehaviour
{
    private float time;

    public GameObject mainMarker;
    public GameObject enemy1;

    private static LinkedList<GameObject> enemies;

    // Start is called before the first frame update
    void Start()
    {
        time = 0;
        enemies = new LinkedList<GameObject>();
    }

    // Update is called once per frame
    void Update()
    {
        time += Time.deltaTime;
        if (time > 1f)
        {
            Vector2 circleRandom = Random.insideUnitCircle.normalized * Random.Range(10, 15);
            Vector3 pos = new Vector3(circleRandom.x, 0.1f, circleRandom.y);
            var obj = Instantiate(enemy1, pos, Quaternion.identity);
            obj.transform.SetParent(mainMarker.transform);
            time = 0;
            enemies.AddFirst(obj);
        }
    }

    public static void deleteEnemy(GameObject enemy)
    {
        enemies.Remove(enemy);
    }
    public static LinkedList<GameObject> getEnemies()
    {
        return enemies;
    }
}
