package com.example.cfm.ui;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.app.AppOpsManager;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.location.Location;
import android.os.Build;
import android.os.Bundle;
import android.os.Looper;
import android.provider.Settings;
import android.view.animation.Animation;
import android.view.animation.AnimationUtils;
import android.widget.ImageView;
import android.widget.LinearLayout;

import com.example.cfm.R;
import com.fonfon.geohash.GeoHash;
import com.google.android.gms.location.FusedLocationProviderClient;
import com.google.android.gms.location.LocationCallback;
import com.google.android.gms.location.LocationRequest;
import com.google.android.gms.location.LocationResult;
import com.google.android.gms.location.LocationServices;

public class SplashActivity extends AppCompatActivity {


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        System.out.println("poo poo poo");

        setContentView(R.layout.activity_splash);

        startAnimation();
    }

    @RequiresApi(api = Build.VERSION_CODES.M)
    @Override
    protected void onResume() {
        super.onResume();
        System.out.println("resumed the thing");
        locationTest();
    }

    @RequiresApi(api = Build.VERSION_CODES.M)
    private void locationTest() {
        FusedLocationProviderClient client = LocationServices.getFusedLocationProviderClient(this);
        System.out.println("permissions status: " + ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION));
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            System.out.println("mango smoothie");
            checkPermissions(AppOpsManager.OPSTR_COARSE_LOCATION, Settings.ACTION_LOCATION_SOURCE_SETTINGS);
        }
        /*client.getCurrentLocation(102, null).addOnSuccessListener(this, new OnSuccessListener<Location>() {
            @Override
            public void onSuccess(Location location) {printLocation(location);}
        });*/
        LocationRequest lr = LocationRequest.create();
        lr.setInterval(500)
                .setFastestInterval(0)
                .setMaxWaitTime(0)
                .setSmallestDisplacement(0)
                .setPriority(LocationRequest.PRIORITY_HIGH_ACCURACY);
        client.requestLocationUpdates(lr, new LocationCallback() {
            @Override
            public void onLocationResult(LocationResult locationResult) {
                if (locationResult == null) {
                    return;
                }
                for (Location location : locationResult.getLocations()) {
                    printLocation(location);
                }
            }
        }, Looper.getMainLooper());

    }

    @RequiresApi(api = Build.VERSION_CODES.M)
    public void checkPermissions(String permission, String setting) {
        System.out.println(permission + "\t" + setting);
        AppOpsManager appOps = (AppOpsManager) getSystemService(Context.APP_OPS_SERVICE);
        if (appOps.checkOpNoThrow(permission, android.os.Process.myUid(), getPackageName()) == AppOpsManager.MODE_ALLOWED)
            System.out.println("we do have permission");
        else startActivityForResult(new Intent(setting), 69);
    }

    private void printLocation(Location location) {
        if (location != null) {
            System.out.println("hell ya nonnull location");
            System.out.println("Location: \t" + location);
            System.out.println("Latitude: \t" + location.getLatitude());
            System.out.println("Longitude:\t" + location.getLongitude());
            GeoHash hash = GeoHash.fromLocation(location, 5);
            System.out.println("Geohash:  \t" + hash);
        } else {
            System.out.println("damn");
        }
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

        Thread splashTread = new Thread() {
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
        splashTread.start();
    }

    private void startLoginActivity() {
        Intent intent = new Intent(SplashActivity.this, LoginActivity.class);
        startActivity(intent);
    }

}