package ui;


import androidx.appcompat.app.AppCompatActivity;
import android.content.Intent;

import android.os.Bundle;
import android.view.animation.Animation;
import android.view.animation.AnimationUtils;
import android.widget.ImageView;
import android.widget.LinearLayout;

import com.example.cfm.R;
<<<<<<< Updated upstream:app/src/main/java/com/example/cfm/ui/SplashActivity.java
=======
import com.fonfon.geohash.GeoHash;
>>>>>>> Stashed changes:app/src/main/java/ui/SplashActivity.java

public class SplashActivity extends AppCompatActivity {


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_splash);

        startAnimation();
    }

    private void startAnimation(){
        Animation anim = AnimationUtils.loadAnimation(this, R.anim.fade);
        anim.reset();
        LinearLayout l = (LinearLayout)findViewById(R.id.lin_layout);
        l.clearAnimation();
        l.startAnimation(anim);

        anim = AnimationUtils.loadAnimation(this, R.anim.translate);
        anim.reset();
        ImageView iv = (ImageView)findViewById(R.id.splash);
        iv.clearAnimation();
        iv.startAnimation(anim);

        Thread splashThread = new Thread() {
            @Override
            public void run() {
                try {
                    int waited = 0;
                    // Splash screen pause time
                    while (waited < 1500) {
                        sleep(200);
                        waited += 200;
                    }

                    overridePendingTransition(R.anim.pull_from_bottom, R.anim.wait);

                    startLoginActivity();
                } catch (InterruptedException e) {
                    // do nothing
                } finally {
                    SplashActivity.this.finish();
                }

            }
        };

        splashThread.start();
    }

    private void startLoginActivity() {
        Intent intent = new Intent(SplashActivity.this, LoginActivity.class);
        startActivity(intent);
    }

}